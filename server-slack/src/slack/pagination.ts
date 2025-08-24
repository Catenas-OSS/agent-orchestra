export async function paginateCursor<T>(
  call: (cursor?: string) => Promise<{ items: T[]; next_cursor?: string }>,
  all: boolean,
  limit?: number
): Promise<T[]> {
  let cursor: string | undefined;
  const results: T[] = [];
  while (true) {
    const { items, next_cursor } = await call(cursor);
    results.push(...items);
    if (!all || !next_cursor) break;
    cursor = next_cursor;
    if (limit && results.length >= limit) break;
  }
  return results;
}
