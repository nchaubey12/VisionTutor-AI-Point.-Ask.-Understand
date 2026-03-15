/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false, // Disabled: StrictMode double-mounts cause WebSocket flapping in dev
  env: {
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL,
  },
};

module.exports = nextConfig;
