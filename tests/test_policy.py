"""Tests for policy module."""

import pytest
import time

from agent_orchestra.policy import (
    Budget, Policy, PolicyRule, PolicyAction, HITLManager,
    create_default_policy, create_default_budget
)


def test_budget_creation():
    """Test Budget creation and basic functionality."""
    budget = Budget(
        max_tokens=1000,
        max_cost=10.0,
        max_children=5,
        max_time=60
    )
    
    assert budget.max_tokens == 1000
    assert budget.max_cost == 10.0
    assert budget.max_children == 5
    assert budget.max_time == 60
    
    # Initial usage should be zero
    assert budget.used_tokens == 0
    assert budget.used_cost == 0.0
    assert budget.used_children == 0


def test_budget_checks():
    """Test budget limit checking."""
    budget = Budget(max_tokens=100, max_cost=5.0, max_children=3, max_time=10)
    
    # Within limits
    assert budget.check_tokens(50) is True
    assert budget.check_cost(2.5) is True
    assert budget.check_children(2) is True
    assert budget.check_time() is True
    
    # Exceeding limits
    assert budget.check_tokens(150) is False
    assert budget.check_cost(7.0) is False
    assert budget.check_children(5) is False


def test_budget_consumption():
    """Test budget consumption tracking."""
    budget = Budget(max_tokens=100, max_cost=5.0, max_children=3)
    
    # Consume resources
    budget.consume_tokens(30)
    budget.consume_cost(2.0)
    budget.consume_children(1)
    
    assert budget.used_tokens == 30
    assert budget.used_cost == 2.0
    assert budget.used_children == 1
    
    # Check remaining capacity
    assert budget.check_tokens(70) is True
    assert budget.check_tokens(71) is False
    assert budget.check_cost(3.0) is True
    assert budget.check_cost(3.1) is False


def test_budget_time_exceeded():
    """Test time budget checking."""
    budget = Budget(max_time=1)  # 1 second
    
    # Initially within time
    assert budget.check_time() is True
    assert not budget.is_exceeded()
    
    # Simulate time passing
    time.sleep(1.1)
    
    # Should now be exceeded
    assert budget.check_time() is False
    assert budget.is_exceeded()


def test_budget_unlimited():
    """Test budget with None (unlimited) values."""
    budget = Budget()  # All None = unlimited
    
    # Should allow any amount
    assert budget.check_tokens(999999) is True
    assert budget.check_cost(999999.0) is True
    assert budget.check_children(999999) is True
    assert budget.check_time() is True


def test_policy_rule_creation():
    """Test PolicyRule creation and evaluation."""
    rule = PolicyRule(
        name="test_rule",
        condition="tool:dangerous_tool",
        action=PolicyAction.DENY,
        reason="Tool is dangerous"
    )
    
    assert rule.name == "test_rule"
    assert rule.action == PolicyAction.DENY
    
    # Test evaluation
    assert rule.evaluate({"tool": "dangerous_tool"}) is True
    assert rule.evaluate({"tool": "safe_tool"}) is False


def test_policy_rule_conditions():
    """Test different policy rule conditions."""
    # Tool condition
    tool_rule = PolicyRule("tool", "tool:file_delete", PolicyAction.DENY)
    assert tool_rule.evaluate({"tool": "file_delete"}) is True
    assert tool_rule.evaluate({"tool": "web_search"}) is False
    
    # Domain condition
    domain_rule = PolicyRule("domain", "domain:evil.com", PolicyAction.DENY)
    assert domain_rule.evaluate({"url": "https://evil.com/bad"}) is True
    assert domain_rule.evaluate({"url": "https://good.com/safe"}) is False
    
    # Untrusted condition
    trust_rule = PolicyRule("trust", "untrusted", PolicyAction.REQUIRE_APPROVAL)
    assert trust_rule.evaluate({"trusted": False}) is True
    assert trust_rule.evaluate({"trusted": True}) is False
    assert trust_rule.evaluate({}) is True  # Default to untrusted


def test_policy_evaluation():
    """Test Policy evaluation logic."""
    policy = Policy()
    policy.require_approval_for_untrusted = False  # Disable for this test
    
    # Set up allowed tools
    policy.set_allowed_tools({"web_search", "analysis"})
    
    # Test allowed tool
    assert policy.evaluate({"tool": "web_search"}) == PolicyAction.ALLOW
    
    # Test disallowed tool
    assert policy.evaluate({"tool": "file_delete"}) == PolicyAction.DENY
    
    # Test no tool specified (should allow)
    assert policy.evaluate({}) == PolicyAction.ALLOW


def test_policy_domain_whitelist():
    """Test domain whitelist functionality."""
    policy = Policy()
    policy.require_approval_for_untrusted = False  # Disable for this test
    policy.set_allowed_domains({"trusted.com", "safe.org"})
    
    # Test allowed domain
    assert policy.evaluate({"url": "https://trusted.com/page"}) == PolicyAction.ALLOW
    
    # Test disallowed domain
    assert policy.evaluate({"url": "https://evil.com/bad"}) == PolicyAction.DENY
    
    # Test no URL (should allow)
    assert policy.evaluate({}) == PolicyAction.ALLOW


def test_policy_untrusted_content():
    """Test untrusted content handling."""
    policy = Policy()
    policy.require_approval_for_untrusted = True
    
    # Test trusted content
    assert policy.evaluate({"trusted": True}) == PolicyAction.ALLOW
    
    # Test untrusted content
    assert policy.evaluate({"trusted": False}) == PolicyAction.REQUIRE_APPROVAL
    
    # Test default (untrusted)
    assert policy.evaluate({}) == PolicyAction.REQUIRE_APPROVAL


def test_policy_custom_rules():
    """Test custom policy rules."""
    policy = Policy()
    policy.require_approval_for_untrusted = False  # Disable for this test
    
    # Add custom rule
    policy.add_rule(PolicyRule(
        name="block_exec",
        condition="tool:system_exec",
        action=PolicyAction.DENY,
        reason="System execution not allowed"
    ))
    
    # Test custom rule
    assert policy.evaluate({"tool": "system_exec"}) == PolicyAction.DENY
    assert policy.evaluate({"tool": "web_search"}) == PolicyAction.ALLOW


def test_policy_budget_check():
    """Test policy budget checking."""
    policy = Policy()
    budget = Budget(max_tokens=100, max_cost=5.0)
    
    # Within budget
    assert policy.check_budget(budget, {"tokens": 50, "cost": 2.0}) is True
    
    # Exceeding token budget
    assert policy.check_budget(budget, {"tokens": 150, "cost": 2.0}) is False
    
    # Exceeding cost budget
    assert policy.check_budget(budget, {"tokens": 50, "cost": 7.0}) is False


@pytest.mark.asyncio
async def test_hitl_manager():
    """Test HITLManager functionality."""
    hitl = HITLManager()
    
    # Request approval
    await hitl.request_approval(
        approval_id="test-approval",
        context={"tool": "dangerous_action"},
        reason="Action requires approval"
    )
    
    # Check pending approvals
    pending = hitl.get_pending_approvals()
    assert "test-approval" in pending
    assert pending["test-approval"]["reason"] == "Action requires approval"
    
    # Check approval status (should be None = pending)
    assert hitl.is_approved("test-approval") is None
    
    # Approve
    success = hitl.approve("test-approval", "admin", "Looks safe")
    assert success is True
    assert hitl.is_approved("test-approval") is True
    
    # Should no longer be pending
    pending = hitl.get_pending_approvals()
    assert "test-approval" not in pending


@pytest.mark.asyncio
async def test_hitl_manager_deny():
    """Test HITLManager denial functionality."""
    hitl = HITLManager()
    
    await hitl.request_approval("test-deny", {}, "Test denial")
    
    # Deny the request
    success = hitl.deny("test-deny", "admin", "Too risky")
    assert success is True
    assert hitl.is_approved("test-deny") is False


def test_default_policy():
    """Test default policy creation."""
    policy = create_default_policy()
    
    # Should block dangerous tools (even if trusted)
    assert policy.evaluate({"tool": "file_delete", "trusted": True}) == PolicyAction.DENY
    assert policy.evaluate({"tool": "system_exec", "trusted": True}) == PolicyAction.DENY
    
    # Should require approval for untrusted content
    assert policy.evaluate({"trusted": False}) == PolicyAction.REQUIRE_APPROVAL
    
    # Should allow safe operations when trusted
    assert policy.evaluate({"tool": "web_search", "trusted": True}) == PolicyAction.ALLOW


def test_default_budget():
    """Test default budget creation."""
    budget = create_default_budget()
    
    assert budget.max_tokens == 1000000
    assert budget.max_cost == 50.0
    assert budget.max_children == 1000
    assert budget.max_time == 3600