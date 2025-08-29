import fs from "fs";
import path from "path";

const SPEC_URL = "https://api.slack.com/specs/openapi/v2/openapi.json";

export async function fetchSpec(): Promise<any> {
  try {
    const res = await fetch(SPEC_URL);
    if (res.ok) return res.json();
  } catch {
    // ignore
  }
  const p = path.resolve(path.dirname(new URL(import.meta.url).pathname), "./static/slack-methods.json");
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}
