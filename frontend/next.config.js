/** @type {import('next').NextConfig} */
const nextConfig = {
  // Produces a self-contained server bundle (.next/standalone) for a lean
  // Docker image — see frontend/Dockerfile.
  output: "standalone",
  async rewrites() {
    const target = process.env.INTERNAL_API_URL || "http://api:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${target}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
