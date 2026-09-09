import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  rewrites: async () => {
    // FLASK_DEV_PORT lets a second checkout (a worktree) run its own backend
    // while another dev server already holds 5328. Unset = the usual 5328, so
    // nothing changes for a single checkout.
    const flaskPort = process.env.FLASK_DEV_PORT || '5328'
    return [
      {
        source: '/api/:path*',
        destination:
          process.env.NODE_ENV === 'development'
            ? `http://127.0.0.1:${flaskPort}/api/:path*`
            : '/api/',
      },
    ]
  },
  turbopack: {
    root: __dirname,
    rules: {
      '**/*.svg': {
        "loaders": ["@svgr/webpack"],
        "as": "*.js"
      },
      '**/*.html': {
        "loaders": ["html-loader"],
        "as": "*.js"
      }
    }
  },
  webpack: (config) => {
    config.module.rules.push({
      test: /\.svg$/,
      use: ["@svgr/webpack"],
    });
    config.module.rules.push({
      test: /\.html$/,
      use: "html-loader",
    });
    return config;
  },
    reactCompiler: true,
};

export default nextConfig;