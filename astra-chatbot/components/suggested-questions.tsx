"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { MessageSquare, TrendingUp, DollarSign, FileSearch, BarChart } from "lucide-react"
import type { SuggestedQuery } from "@/lib/types"

interface SuggestedQuestionsProps {
  queries: SuggestedQuery[]
  onQuerySelect: (query: string) => void
  disabled: boolean
}

export function SuggestedQuestions({ queries, onQuerySelect, disabled }: SuggestedQuestionsProps) {
  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case "financial analysis":
        return <DollarSign className="h-3 w-3" />
      case "budget overview":
        return <BarChart className="h-3 w-3" />
      case "procurement analysis":
        return <FileSearch className="h-3 w-3" />
      case "trend analysis":
        return <TrendingUp className="h-3 w-3" />
      default:
        return <MessageSquare className="h-3 w-3" />
    }
  }

  const getCategoryColor = (category: string) => {
    switch (category.toLowerCase()) {
      case "financial analysis":
        return "bg-chart-2/10 text-chart-2 border-chart-2/20"
      case "budget overview":
        return "bg-accent/10 text-accent border-accent/20"
      case "procurement analysis":
        return "bg-chart-4/10 text-chart-4 border-chart-4/20"
      case "trend analysis":
        return "bg-chart-1/10 text-chart-1 border-chart-1/20"
      default:
        return "bg-muted text-muted-foreground border-border"
    }
  }

  if (disabled) {
    return (
      <div className="text-center py-8 space-y-2">
        <p className="text-sm text-muted-foreground">Select a dataset first</p>
        <p className="text-xs text-muted-foreground">Choose a dataset to see relevant suggested questions</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {queries.map((query) => (
        <div key={query.id} className="space-y-2">
          <Badge className={getCategoryColor(query.category)} variant="outline">
            {getCategoryIcon(query.category)}
            <span className="ml-1 text-xs">{query.category}</span>
          </Badge>
          <Button
            variant="outline"
            className="w-full text-left h-auto p-3 justify-start bg-transparent hover:bg-accent/5 hover:border-accent/30"
            onClick={() => onQuerySelect(query.text)}
          >
            <MessageSquare className="h-4 w-4 mr-2 flex-shrink-0 text-muted-foreground" />
            <span className="text-sm text-pretty text-foreground">{query.text}</span>
          </Button>
        </div>
      ))}
    </div>
  )
}
