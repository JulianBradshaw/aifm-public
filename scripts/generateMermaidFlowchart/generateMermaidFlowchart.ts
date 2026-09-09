import { mermaidConfig } from './modelDiagram.cfg';
import fs from 'fs/promises';

async function main() {

  const { run } = await import('@mermaid-js/mermaid-cli');

  const topDownModelDiagram = await fs.readFile("scripts/generateMermaidFlowchart/modelDiagram.mmd", "utf8");
  await fs.writeFile("scripts/generateMermaidFlowchart/modelDiagramTopDown.mmd", topDownModelDiagram.replace("flowchart LR", "flowchart TD"));

  run("scripts/generateMermaidFlowchart/modelDiagram.mmd", "svgs/modelDiagram.svg", {
    parseMMDOptions: {
      mermaidConfig,
      backgroundColor: "#fffff8",
    },
  });

  run("scripts/generateMermaidFlowchart/modelDiagramTopDown.mmd", "svgs/modelDiagramTopDown.svg", {
    parseMMDOptions: {
      mermaidConfig,
      backgroundColor: "#fffff8",
    },
  });
}

main();