"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import { AuditTrail } from "@/components/audit-trail"
import { ScreenReaderOnly } from "@/components/accessibility/screen-reader-only"
import { Shield, ChevronDown, ChevronUp, Clock, Target } from "lucide-react"
import type { QueryResult } from "@/lib/types"

interface ResponsePanelProps {
  result: QueryResult
  isLatest: boolean
}

export function ResponsePanel({ result, isLatest }: ResponsePanelProps) {
  const [isAuditOpen, setIsAuditOpen] = useState(isLatest)

  const getTrustColor = (trustScore: string) => {
    switch (trustScore) {
      case "high":
        return "text-chart-2 bg-chart-2/10 border-chart-2/20"
      case "medium":
        return "text-accent bg-accent/10 border-accent/20"
      case "low":
        return "text-destructive bg-destructive/10 border-destructive/20"
      default:
        return "text-muted-foreground bg-muted border-border"
    }
  }

  const getTrustLabel = (trustScore: string) => {
    switch (trustScore) {
      case "high":
        return "High Trust"
      case "medium":
        return "Medium Trust"
      case "low":
        return "Low Trust"
      default:
        return "Unknown"
    }
  }

  return (
    <Card
      className={isLatest ? "ring-2 ring-accent/20 border-accent/30" : ""}
      role="article"
      aria-labelledby={`query-${result.id}-title`}
    >
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-2 flex-1">
            <CardTitle id={`query-${result.id}-title`} className="text-lg text-pretty">
              {result.query}
            </CardTitle>
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <div className="flex items-center space-x-1">
                <Clock className="h-3 w-3" aria-hidden="true" />
                <span>
                  <ScreenReaderOnly>Query submitted at </ScreenReaderOnly>
                  {new Date(result.timestamp).toLocaleString()}
                </span>
              </div>
              <div className="flex items-center space-x-1">
                <Target className="h-3 w-3" aria-hidden="true" />
                <span>
                  <ScreenReaderOnly>Confidence level: </ScreenReaderOnly>
                  {result.confidence}% confidence
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Badge
              className={getTrustColor(result.trustScore)}
              aria-label={`Trust level: ${getTrustLabel(result.trustScore)}`}
            >
              <Shield className="mr-1 h-3 w-3" aria-hidden="true" />
              {getTrustLabel(result.trustScore)}
            </Badge>
            {isLatest && (
              <Badge variant="secondary" className="text-xs" aria-label="Most recent query result">
                Latest
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Response */}
        <div className="space-y-2">
          <h4 className="font-medium text-foreground">Response</h4>
          <div className="p-4 bg-muted/50 rounded-md" role="region" aria-label="AI response">
            <p className="text-foreground text-pretty leading-relaxed">{result.answer}</p>
          </div>
        </div>

        <Separator />

        {/* Audit Trail Toggle */}
        <Collapsible open={isAuditOpen} onOpenChange={setIsAuditOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className="w-full justify-between p-0 h-auto"
              aria-expanded={isAuditOpen}
              aria-controls={`audit-trail-${result.id}`}
            >
              <span className="font-medium text-foreground">View Audit Trail & Sources</span>
              {isAuditOpen ? (
                <ChevronUp className="h-4 w-4" aria-hidden="true" />
              ) : (
                <ChevronDown className="h-4 w-4" aria-hidden="true" />
              )}
              <ScreenReaderOnly>{isAuditOpen ? "Hide" : "Show"} detailed audit trail and data sources</ScreenReaderOnly>
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent
            className="space-y-4 pt-4"
            id={`audit-trail-${result.id}`}
            role="region"
            aria-label="Audit trail and data sources"
          >
            <AuditTrail result={result} />
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  )
}
