import fs from "fs";
import path from "path";
import Handlebars from "handlebars";
import { fetchSpec } from "./fetchSpec.js";
import { parseSpec, MethodSpec } from "./parseSpec.js";

function mapType(t: string) {
  switch (t) {
    case "string":
      return "string";
    case "number":
      return "number";
    case "boolean":
      return "boolean";
    default:
      return "any";
  }
}

async function generate() {
  const raw = await fetchSpec();
  const methods = parseSpec(raw);
  const template = Handlebars.compile(fs.readFileSync(path.join("codegen/templates/tool.hbs"), "utf-8"));

  const outDir = path.join("src/tools/generated");
  fs.mkdirSync(outDir, { recursive: true });
  const imports: string[] = [];
  for (const m of methods) {
    const params = m.params.map(p => ({ name: p.name, required: p.required, tsType: mapType(p.type) }));
    const content = template({ methodName: m.name, description: m.description, params });
    const file = path.join(outDir, `${m.name.replace(/\./g, '.')}.ts`);
    fs.writeFileSync(file, content);
    imports.push(`import * as ${m.name.replace(/\./g, '_')} from "./generated/${m.name}.js";`);
  }
  // update registry
  const registryPath = path.join("src/tools/registry.ts");
  const registryContent = `import { Server } from "@modelcontextprotocol/typescript-sdk";\nimport * as genericCall from "./genericCall.js";\n${imports.join('\n')}\nconst tools = [genericCall, ${methods.map(m => m.name.replace(/\./g, '_')).join(', ')}];\nexport async function registerAllTools(server: Server){\n for(const t of tools){ server.tool({name:(t as any).name, description:(t as any).description, inputSchema:(t as any).inputSchema, outputSchema:(t as any).outputSchema, handler:(t as any).handler}); }\n}\n`;
  fs.writeFileSync(registryPath, registryContent);
}

generate().catch(err => {
  console.error(err);
  process.exit(1);
});
