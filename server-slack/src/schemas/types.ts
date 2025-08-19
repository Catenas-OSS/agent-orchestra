import { z } from "zod";

export const cursor = z.string().describe("Pagination cursor");
