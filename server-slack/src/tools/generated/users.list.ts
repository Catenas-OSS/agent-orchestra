import { z } from "zod";
import { resolveSlackToken } from "../../auth/token.js";
import { callSlack } from "../../slack/client.js";
import { paginateCursor } from "../../slack/pagination.js";

export const name = "slack.users.list";
export const description = "List users in a workspace";

export const inputSchema = z.object({
  limit: z.number().int().optional(),
  cursor: z.string().optional(),
  all: z.boolean().optional(),
});

export type Input = z.infer<typeof inputSchema>;

export async function handler(ctx: any, input: Input) {
  const token = resolveSlackToken(ctx);
  const callPage = async (cursor?: string) => {
    const params: any = {};
    if (input.limit) params.limit = input.limit;
    if (cursor) params.cursor = cursor;
    const resp: any = await callSlack(token, "users.list", params);
    return {
      items: resp.members || [],
      next_cursor: resp.response_metadata?.next_cursor,
    };
  };
  if (input.all) {
    const members = await paginateCursor(callPage, true, input.limit);
    return { members };
  }
  const resp = await callPage(input.cursor);
  return { members: resp.items, next_cursor: resp.next_cursor };
}
