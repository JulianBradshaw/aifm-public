import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      screens: {
        'md': '930px',
      },
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        'dull-background': 'var(--dull-background)',
        'dull-foreground': 'var(--dull-foreground)',
        'vivid-background': 'var(--vivid-background)',
        'vivid-foreground': 'var(--vivid-foreground)',
        accent: {
          DEFAULT: 'var(--accent-color)',
          foreground: 'var(--accent-foreground)'
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))'
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))'
        },
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))'
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))'
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))'
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))'
        },
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        chart: {
          '1': 'hsl(var(--chart-1))',
          '2': 'hsl(var(--chart-2))',
          '3': 'hsl(var(--chart-3))',
          '4': 'hsl(var(--chart-4))',
          '5': 'hsl(var(--chart-5))'
        }
      },
      fontFamily: {
        'system-ui': [
          '-apple-system',
          'BlinkMacSystemFont',
          'avenir next',
          'avenir',
          'segoe ui',
          'helvetica neue',
          'helvetica',
          'cantarell',
          'ubuntu',
          'roboto',
          'noto',
          'arial',
          'sans-serif'
        ],
        'system-mono': [
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'Monaco',
          'Liberation Mono',
          'Lucida Console',
          'monospace'
        ],
        'et-book': [
          'et-book',
          'serif'
        ],
        'et-book-italic': [
          'et-book-italic',
          'serif'
        ],
        'et-book-bold': [
          'et-book-bold',
          'serif'
        ]
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)'
      },
    },
  },
  plugins: [],
}

export default config