/** @type {import('next').NextConfig} */
const nextConfig = {
  // Produces a self-contained server bundle (.next/standalone) for a lean
  // Docker image — see frontend/Dockerfile.
  output: "standalone",
};

module.exports = nextConfig;
