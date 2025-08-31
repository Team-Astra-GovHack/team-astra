import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { FileText, Database, CheckCircle, ArrowRight } from "lucide-react"
import type { QueryResult } from "@/lib/types"

interface AuditTrailProps {
  result: QueryResult
}

export function AuditTrail({ result }: AuditTrailProps) {
  return (
    <div className="space-y-4">
      {/* Data Sources */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Database className="h-4 w-4 text-accent" />
            <CardTitle className="text-base">Data Sources</CardTitle>
          </div>
          <CardDescription>Datasets and sources used for this analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {result.sources.map((source, index) => (
              <div key={index} className="flex items-center space-x-2 p-2 bg-muted/30 rounded">
                <FileText className="h-3 w-3 text-muted-foreground" />
                <span className="text-sm text-foreground">{source}</span>
                <Badge variant="outline" className="ml-auto text-xs">
                  Verified
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Reasoning Steps */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-4 w-4 text-accent" />
            <CardTitle className="text-base">Analysis Steps</CardTitle>
          </div>
          <CardDescription>How we processed your query and generated the response</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {result.reasoning.map((step, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-6 h-6 bg-accent/10 text-accent rounded-full flex items-center justify-center text-xs font-medium">
                  {index + 1}
                </div>
                <div className="flex-1 space-y-1">
                  <p className="text-sm text-foreground text-pretty">{step}</p>
                  {index < result.reasoning.length - 1 && (
                    <div className="flex items-center space-x-2 py-1">
                      <ArrowRight className="h-3 w-3 text-muted-foreground" />
                      <div className="h-px bg-border flex-1"></div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Trust Score Details */}
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-4 w-4 text-accent" />
            <CardTitle className="text-base">Trust Assessment</CardTitle>
          </div>
          <CardDescription>Factors contributing to the trust score</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Data Source Reliability</span>
              <Badge variant="outline" className="text-xs">
                {result.trustScore === "high" ? "Excellent" : result.trustScore === "medium" ? "Good" : "Limited"}
              </Badge>
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Cross-Reference Validation</span>
              <Badge variant="outline" className="text-xs">
                {result.sources.length > 2 ? "Multiple Sources" : "Single Source"}
              </Badge>
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Confidence Level</span>
              <Badge variant="outline" className="text-xs">
                {result.confidence}%
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
