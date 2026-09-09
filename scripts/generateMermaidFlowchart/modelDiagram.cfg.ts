import type { MermaidConfig } from "mermaid";

export const mermaidConfig: MermaidConfig = {
  "theme": "base",
  "look": "handDrawn",
  "layout": "elk",
  "elk": {
    "mergeEdges": true,
    // "nodePlacementStrategy": "",
    "cycleBreakingStrategy": "GREEDY_MODEL_ORDER"
  },
  "themeVariables": {
    "background": "#fffff8",
    "primaryTextColor": "#171717",
    "primaryColor": "#fffff8",
    // "primaryBorderColor": "#fffff8",
    "secondaryColor": "#fffff8",
    "tertiaryColor": "#fffff8",
    // "tertiaryBorderColor": "#ffffff",
    "lineColor": "#2A623D",
    "borderColor": "#2A623D",
    "fontFamily": "SFMono-Regular, Menlo, Consolas, Monaco, Liberation Mono, Lucida Console, monospace"
  }
}