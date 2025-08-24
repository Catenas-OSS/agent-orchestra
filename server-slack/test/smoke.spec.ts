import { describe, it, expect } from "vitest";
import { handler as chatPostMessage } from "../src/tools/generated/chat.postMessage";

describe("chat.postMessage", () => {
  it("requires token", async () => {
    await expect(chatPostMessage({}, { channel: "C1", text: "hi" })).rejects.toThrow();
  });
});
