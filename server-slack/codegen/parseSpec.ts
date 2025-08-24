export interface Param {
  name: string;
  type: string;
  required: boolean;
}
export interface MethodSpec {
  name: string;
  description: string;
  params: Param[];
  cursorPagination?: boolean;
  isFileUpload?: boolean;
}

export function parseSpec(raw: any): MethodSpec[] {
  if (Array.isArray(raw)) return raw as MethodSpec[];
  // minimal openapi parsing
  const methods: MethodSpec[] = [];
  const paths = raw.paths || {};
  for (const [url, item] of Object.entries<any>(paths)) {
    const methodName = url.replace(/^\//, "");
    const post = item.post || item.get;
    const params: Param[] = (post.parameters || []).map((p: any) => ({
      name: p.name,
      type: p.schema?.type || "string",
      required: p.required || false,
    }));
    methods.push({ name: methodName, description: post.summary || "", params });
  }
  return methods;
}
