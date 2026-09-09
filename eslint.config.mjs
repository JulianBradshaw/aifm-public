import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

// eslint-config-next@16 ships native flat config (arrays of config objects), so
// they're spread in directly. Do NOT route these through FlatCompat — that's the
// legacy-eslintrc→flat adapter, and feeding it already-flat config makes the
// eslintrc schema validator throw "wrong type (expected object but got [array])".
const eslintConfig = [
  ...nextCoreWebVitals,
  ...nextTypescript,
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      ".venv/**",
      "out/**",
      "build/**",
      "next-env.d.ts",
      // Git worktrees live here. Without this, lint crawls every worktree and
      // reports other branches' errors as if they were this tree's — 2,933
      // duplicated source files against 142 real ones, and minutes per run.
      ".claude/**",
    ],
  },
];

export default eslintConfig;
