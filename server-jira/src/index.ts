import express from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp';
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types';
import fetch, { RequestInit } from 'node-fetch';
import { z } from 'zod';
import crypto from 'node:crypto';

interface GetIssueInput {
  issueIdOrKey: string;
}

interface SearchIssuesInput {
  jql: string;
  cursor?: number;
  limit?: number;
  all?: boolean;
}

interface CreateIssueInput {
  projectKey: string;
  summary: string;
  description?: string;
  issueType?: string;
}

interface ListProjectsInput {
  cursor?: number;
  limit?: number;
  all?: boolean;
}


// Load config from environment
const PORT = process.env.PORT ? parseInt(process.env.PORT) : 4000;
const JIRA_BASE_URL = process.env.JIRA_BASE_URL || '';

// e.g. "https://your-domain.atlassian.net"
if (!JIRA_BASE_URL) {
  console.error("Error: JIRA_BASE_URL environment variable is not set.");
  process.exit(1);
}

// Utility: create authorized headers for Jira API calls
function jiraAuthHeaders(token: string) {
  return {
    'Authorization': `Bearer ${token}`,
    'Accept': 'application/json',
    'Content-Type': 'application/json'
  };
}

// Utility: Perform a Jira API request with proper headers and error handling
async function jiraRequest(token: string, path: string, init: RequestInit = {}): Promise<any> {
  const url = `${JIRA_BASE_URL}${path}`;
  const headers = { ...(init.headers || {}), ...jiraAuthHeaders(token) };
  try {
    const response = await fetch(url, { ...init, headers });
    if (!response.ok) {
      // Attempt to parse error details from Jira
      let errorData: any;
      try {
        errorData = await response.json();
      } catch {
        errorData = { errorMessages: [`HTTP ${response.status}`] };
      }
      // Compose error message from Jira's error response
      let message = `Jira API Error (HTTP ${response.status})`;
      if (errorData.errorMessages) {
        message += `: ${errorData.errorMessages.join('; ')}`;
      } else if (errorData.message) {
        message += `: ${errorData.message}`;
      }
      // Throw an error object that MCP will convert to error response
      throw { code: response.status, message };
    }
    // Return parsed JSON for success responses
    if (response.status === 204) {
      // No content
      return null;
    }
    return await response.json();
  } catch (err: any) {
    // If we threw an object with code/message, rethrow it for MCP error handling
    if (err && typeof err.code === 'number') {
      throw err;
    }
    // Otherwise, throw a generic error
    throw { code: -32000, message: err?.message || 'Unknown error' };
  }
}

// Setup Express and MCP transport
const app = express();
app.use(express.json());

// In-memory store for active transports by session
const transports: { [sessionId: string]: StreamableHTTPServerTransport } = {};

// **Authentication middleware** for MCP route: ensure token is provided
app.use('/mcp', (req, res, next) => {
  const authHeader = req.get('Authorization');
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    // Return JSON-RPC error for missing token
    res.status(401).json({
      jsonrpc: '2.0',
      error: { code: -32602, message: 'Authentication required: Jira API token missing or invalid' },
      id: req.body?.id ?? null
    });
    return;
  }
  next();
});

// Handle MCP POST requests (client-to-server JSON-RPC calls)
app.post('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'] as string | undefined;
  let transport: StreamableHTTPServerTransport;
  if (sessionId && transports[sessionId]) {
    // Reuse existing session transport
    transport = transports[sessionId];
  } else if (!sessionId && isInitializeRequest(req.body)) {
    // Start a new MCP session
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => crypto.randomUUID(),  // generate unique session IDs
      onsessioninitialized: (newSessionId: string) => {
        // Store transport for this session
        transports[newSessionId] = transport;
      }
    });
    // Clean up on session close
    transport.onclose = () => {
      if (transport.sessionId) {
        delete transports[transport.sessionId];
      }
    };
    // **Create a new MCP server instance for this session**
    const mcpServer = new McpServer({ name: 'JiraMCP', version: '1.0.0' });

    // Register all Jira tools on this server
    // (We'll illustrate a few examples here. The code generation should continue similarly for all endpoints.)

    // Example 1: Get Issue details (GET /rest/api/2/issue/{issueIdOrKey})
    mcpServer.registerTool(
      "jira.issue.getIssue",
      {
        description: "Get details of a Jira issue by ID or key",
        inputSchema: {
          issueIdOrKey: z.string().describe("Issue ID or issue key (e.g. PROJ-123)")
        }
      },
      async (input: GetIssueInput, requestContext: any) => {
        const { issueIdOrKey } = input;
        const token = String(requestContext.headers['authorization'] || '').replace(/^Bearer\s+/, '');
        const path = `/rest/api/2/issue/${encodeURIComponent(issueIdOrKey)}`;
        const issue: any = await jiraRequest(token, path);
        return { content: issue } as any;
      }
    );

    // Example 2: Search issues (GET /rest/api/2/search)
    mcpServer.registerTool(
      "jira.issue.searchIssues",
      {
        description: "Search for issues using JQL query",
        inputSchema: {
          jql: z.string().describe("JQL query string"),
          cursor: z.number().optional().describe("Start index for pagination (default 0)"),
          limit: z.number().optional().describe("Max results per page (default 50)"),
          all: z.boolean().optional().describe("If true, retrieve all results by paging through")
        }
      },
      async (input: SearchIssuesInput, requestContext: any) => {
        const { jql, cursor, limit, all } = input;
        const token = String(requestContext.headers['authorization'] || '').replace(/^Bearer\s+/, '');
        const maxResults = limit ?? 50;
        let startAt = cursor ?? 0;
        const results: any[] = [];
        do {
          const query = new URLSearchParams({ jql, startAt: String(startAt), maxResults: String(maxResults) });
          const data: any = await jiraRequest(token, `/rest/api/2/search?${query.toString()}`);
          // Accumulate issues
          if (data.issues) {
            results.push(...data.issues);
          }
          // Stop if not auto-fetching all pages
          if (!all) {
            return { content: data } as any;
          }
          // Otherwise, prepare for next page
          startAt = data.startAt + data.maxResults;
          if (startAt >= data.total) {
            // All issues retrieved
            return { content: { ...data, issues: results, maxResults: data.total, startAt: 0, total: data.total } } as any;
          }
        } while (true);
      }
    );

    // Example 3: Create a new issue (POST /rest/api/2/issue)
    mcpServer.registerTool(
      "jira.issue.createIssue",
      {
        description: "Create a new Jira issue in a project",
        inputSchema: {
          projectKey: z.string().describe("Project key to create the issue in"),
          summary: z.string().describe("Summary/title of the issue"),
          description: z.string().optional().describe("Detailed description of the issue"),
          issueType: z.string().default("Task").describe("Issue type name (e.g. Bug, Task, Story)")
          // Additional fields can be added as needed (priority, assignee, etc.)
        }
      },
      async (input: CreateIssueInput, requestContext: any) => {
        const { projectKey, summary, description, issueType } = input;
        const token = String(requestContext.headers['authorization'] || '').replace(/^Bearer\s+/, '');
        const payload = {
          fields: {
            project: { key: projectKey },
            summary: summary,
            description: description ?? "",
            issuetype: { name: issueType }
          }
        };
        const newIssue: any = await jiraRequest(token, `/rest/api/2/issue`, {
          method: 'POST',
          body: JSON.stringify(payload)
        });
        return { content: newIssue } as any;
      }
    );

    // Example 4: List all projects (GET /rest/api/2/project or /rest/api/2/project/search)
    mcpServer.registerTool(
      "jira.project.listProjects",
      {
        description: "List all projects visible to the user (supports pagination)",
        inputSchema: {
          cursor: z.number().optional().describe("Start index for pagination"),
          limit: z.number().optional().describe("Maximum number of projects to fetch"),
          all: z.boolean().optional().describe("Fetch all projects (ignore pagination)")
        }
      },
      async (input: ListProjectsInput, requestContext: any) => {
        const { cursor, limit, all } = input;
        const token = String(requestContext.headers['authorization'] || '').replace(/^Bearer\s+/, '');
        // Jira has both a non-paginated and a paginated project listing. We'll use the paginated API for consistency.
        let startAt = cursor ?? 0;
        const maxResults = limit ?? 50;
        const allProjects: any[] = [];
        do {
          const query = new URLSearchParams({ startAt: String(startAt), maxResults: String(maxResults) });
          const data: any = await jiraRequest(token, `/rest/api/2/project/search?${query.toString()}`);
          if (data.values) {
            allProjects.push(...data.values);
          }
          if (!all) {
            // Return the single page of results
            return { content: data } as any;
          }
          // If all=true, continue to get all pages
          startAt = data.startAt + data.maxResults;
          if (startAt >= data.total) {
            return { content: { values: allProjects, startAt: 0, maxResults: data.total, total: data.total } } as any;
          }
        } while (true);
      }
    );

    // ... (Continue defining **all other Jira API tools** in a similar manner)
    // For brevity, we have shown a few representative tools.

    // Connect the MCP server to the transport
    await mcpServer.connect(transport);
  } else {
    // Invalid request (no session, not an initialization)
    res.status(400).json({
      jsonrpc: '2.0',
      error: { code: -32000, message: 'Bad Request: No valid session ID provided' },
      id: req.body?.id ?? null
    });
    return;
  }
  // Let the transport handle this request (will invoke the appropriate tool)
  try {
    await transport.handleRequest(req, res, req.body);
  } catch (err) {
    console.error("Error handling MCP request:", err);
    // If an error happened outside of tool execution, send generic failure
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: '2.0',
        error: { code: -32603, message: 'Internal MCP Server error' },
        id: req.body?.id ?? null
      });
    }
  }
});

// Support SSE for server->client messages (if the MCP protocol uses it)
app.get('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'] as string;
  if (!sessionId || !transports[sessionId]) {
    res.status(400).send('Invalid or missing session ID');
    return;
  }
  const transport = transports[sessionId];
  await transport.handleRequest(req, res);
});

// Allow clients to terminate sessions
app.delete('/mcp', async (req, res) => {
  const sessionId = req.headers['mcp-session-id'] as string;
  if (!sessionId || !transports[sessionId]) {
    res.status(400).send('Invalid or missing session ID');
    return;
  }
  const transport = transports[sessionId];
  await transport.handleRequest(req, res);
});

// Start the server
app.listen(PORT, () => {
  console.error(`✅ @modelcontextprotocol/server-jira is running on port ${PORT} and ready at /mcp/`);
});
