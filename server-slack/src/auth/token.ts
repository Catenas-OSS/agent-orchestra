export function resolveSlackToken(ctx: any): string {
  const headerToken = ctx?.headers?.authorization || ctx?.headers?.Authorization;
  let token: string | undefined;
  if (headerToken && typeof headerToken === "string") {
    const match = headerToken.match(/Bearer\s+(.+)/i);
    if (match) token = match[1];
  }
  if (!token) {
    token = process.env.SLACK_TOKEN || process.env.SLACK_BOT_TOKEN;
  }
  if (!token) {
    throw new Error("Slack token required");
  }
  return token;
}
