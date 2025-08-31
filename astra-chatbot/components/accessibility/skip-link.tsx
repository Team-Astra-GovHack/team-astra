import { Button } from "@/components/ui/button"

export function SkipLink() {
  return (
    <Button
      asChild
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-50 bg-accent text-accent-foreground"
    >
      <a href="#main-content">Skip to main content</a>
    </Button>
  )
}
