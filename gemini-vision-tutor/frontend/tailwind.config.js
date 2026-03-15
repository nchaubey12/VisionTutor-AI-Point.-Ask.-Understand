/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./hooks/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0a0a0f",
      },
      animation: {
        "spin-slow": "spin 3s linear infinite",
        scan: "scan 2s linear infinite",
      },
    },
  },
  plugins: [],
};
