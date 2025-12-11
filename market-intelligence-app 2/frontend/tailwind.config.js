/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      colors: {
        gray: {
          900: '#0a0a0f',
          800: '#1a1a24',
          700: '#2a2a38',
        }
      }
    },
  },
  plugins: [],
}
