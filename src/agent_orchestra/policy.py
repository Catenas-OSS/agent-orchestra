"""Policy and budget management.

Implements budgets, HITL gates, and security policies.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PolicyAction(Enum):
    """Actions that can be taken by policies."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class Budget:
    """Resource budget for runs."""

    max_tokens: int | None = None
    max_cost: float | None = None
    max_children: int | None = None
    max_time: int | None = None  # seconds

    # Current usage
    used_tokens: int = 0
    used_cost: float = 0.0
    used_children: int = 0
    start_time: float = field(default_factory=time.time)

    def check_tokens(self, additional_tokens: int) -> bool:
        """Check if additional token usage would exceed budget.
        
        Args:
            additional_tokens: Number of tokens to check against budget.
            
        Returns:
            True if usage would not exceed budget, False otherwise.
        """
        if self.max_tokens is None:
            return True
        return (self.used_tokens + additional_tokens) <= self.max_tokens

    def check_cost(self, additional_cost: float) -> bool:
        """Check if additional cost would exceed budget."""
        if self.max_cost is None:
            return True
        return (self.used_cost + additional_cost) <= self.max_cost

    def check_children(self, additional_children: int = 1) -> bool:
        """Check if additional children would exceed budget."""
        if self.max_children is None:
            return True
        return (self.used_children + additional_children) <= self.max_children

    def check_time(self) -> bool:
        """Check if current runtime exceeds budget."""
        if self.max_time is None:
            return True
        elapsed = time.time() - self.start_time
        return elapsed <= self.max_time

    def consume_tokens(self, tokens: int) -> None:
        """Consume tokens from budget.
        
        Args:
            tokens: Number of tokens to consume.
        """
        self.used_tokens += tokens

    def consume_cost(self, cost: float) -> None:
        """Consume cost from budget."""
        self.used_cost += cost

    def consume_children(self, children: int = 1) -> None:
        """Consume children from budget."""
        self.used_children += children

    def is_exceeded(self) -> bool:
        """Check if any budget limit is exceeded.
        
        Returns:
            True if any budget limit (tokens, cost, children, time) is exceeded.
        """
        return not all([
            self.check_tokens(0),
            self.check_cost(0.0),
            self.check_children(0),
            self.check_time()
        ])


@dataclass
class PolicyRule:
    """A single policy rule."""

    name: str
    condition: str  # Simple string-based condition (could be CEL/Rego in future)
    action: PolicyAction
    reason: str = ""

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate if this rule applies to the given context."""
        # Simple string matching for now
        # In a real implementation, this would use a proper policy language

        if "tool:" in self.condition:
            tool_name = self.condition.split("tool:")[1].strip()
            return context.get("tool") == tool_name

        if "domain:" in self.condition:
            domain = self.condition.split("domain:")[1].strip()
            url = context.get("url", "")
            return domain in url

        if "untrusted" in self.condition:
            return not context.get("trusted", False)

        return False


class Policy:
    """Policy engine for access control and governance."""

    def __init__(self) -> None:
        self.rules: list[PolicyRule] = []
        self.allowed_tools: set[str] | None = None
        self.allowed_domains: set[str] | None = None
        self.require_approval_for_untrusted: bool = True

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule."""
        self.rules.append(rule)

    def set_allowed_tools(self, tools: set[str]) -> None:
        """Set allowed tools whitelist."""
        self.allowed_tools = tools

    def set_allowed_domains(self, domains: set[str]) -> None:
        """Set allowed domains whitelist."""
        self.allowed_domains = domains

    def evaluate(self, context: dict[str, Any]) -> PolicyAction:
        """Evaluate policies against a context and return action."""

        # Evaluate custom rules first (they can deny or require approval)
        for rule in self.rules:
            if rule.evaluate(context):
                return rule.action

        # Check tool whitelist
        if self.allowed_tools is not None:
            tool = context.get("tool")
            if tool and tool not in self.allowed_tools:
                return PolicyAction.DENY

        # Check domain whitelist
        if self.allowed_domains is not None:
            url = context.get("url", "")
            if url and not any(domain in url for domain in self.allowed_domains):
                return PolicyAction.DENY

        # Check untrusted content (only if no explicit trust status and other checks passed)
        if self.require_approval_for_untrusted and not context.get("trusted", False):
            return PolicyAction.REQUIRE_APPROVAL

        # Default allow
        return PolicyAction.ALLOW

    def check_budget(self, budget: Budget, usage: dict[str, Any]) -> bool:
        """Check if usage would exceed budget."""
        tokens = usage.get("tokens", 0)
        cost = usage.get("cost", 0.0)
        children = usage.get("children", 1)

        return all([
            budget.check_tokens(tokens),
            budget.check_cost(cost),
            budget.check_children(children),
            budget.check_time()
        ])


class HITLManager:
    """Human-in-the-loop approval management."""

    def __init__(self) -> None:
        self.pending_approvals: dict[str, dict[str, Any]] = {}

    async def request_approval(
        self,
        approval_id: str,
        context: dict[str, Any],
        reason: str
    ) -> None:
        """Request human approval for an action."""
        self.pending_approvals[approval_id] = {
            "context": context,
            "reason": reason,
            "timestamp": time.time(),
            "status": "pending"
        }

    def approve(self, approval_id: str, approver: str, reason: str = "") -> bool:
        """Approve a pending request."""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id].update({
                "status": "approved",
                "approver": approver,
                "approval_reason": reason,
                "approval_timestamp": time.time()
            })
            return True
        return False

    def deny(self, approval_id: str, approver: str, reason: str = "") -> bool:
        """Deny a pending request."""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id].update({
                "status": "denied",
                "approver": approver,
                "denial_reason": reason,
                "approval_timestamp": time.time()
            })
            return True
        return False

    def get_pending_approvals(self) -> dict[str, dict[str, Any]]:
        """Get all pending approval requests."""
        return {
            k: v for k, v in self.pending_approvals.items()
            if v.get("status") == "pending"
        }

    def is_approved(self, approval_id: str) -> bool | None:
        """Check if an approval request was approved."""
        if approval_id not in self.pending_approvals:
            return None

        status = self.pending_approvals[approval_id].get("status")
        if status == "approved":
            return True
        elif status == "denied":
            return False
        else:
            return None  # Still pending


def create_default_policy() -> Policy:
    """Create a default security policy."""
    policy = Policy()

    # Block dangerous tools by default
    dangerous_tools = {
        "file_write", "file_delete", "system_exec", "network_scan"
    }

    for tool in dangerous_tools:
        policy.add_rule(PolicyRule(
            name=f"block_{tool}",
            condition=f"tool:{tool}",
            action=PolicyAction.DENY,
            reason=f"Tool {tool} is blocked by default security policy"
        ))

    # Require approval for untrusted content
    policy.add_rule(PolicyRule(
        name="untrusted_approval",
        condition="untrusted",
        action=PolicyAction.REQUIRE_APPROVAL,
        reason="Content is not trusted, requires human approval"
    ))

    return policy


def create_default_budget() -> Budget:
    """Create a default budget with reasonable limits."""
    return Budget(
        max_tokens=1000000,  # 1M tokens
        max_cost=50.0,       # $50
        max_children=1000,   # 1000 sub-agents
        max_time=3600        # 1 hour
    )

