import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Shield,
  Database,
  Search,
  FileText,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";
import Image from "next/image";

export default function HomePage() {
  return (
    <div className=" px-4 py-12">
      {/* Hero + Side Sections */}
      <section
        className="
          grid grid-cols-1 lg:grid-cols-12 gap-8 items-start
        "
      >
        {/* Left: Key Features */}
        <div className="order-2 lg:order-1 lg:col-span-3 space-y-6">
          <h2 className="text-2xl font-bold text-foreground">Key Features</h2>
          <p className="text-muted-foreground">
            Built for transparency, accuracy, and accessibility in government
            data analysis
          </p>
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Shield className="h-5 w-5 text-accent" />
                  <CardTitle className="text-lg">Trust Indicators</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Every response includes confidence scores and trust levels to
                  help you assess reliability.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Database className="h-5 w-5 text-accent" />
                  <CardTitle className="text-lg">Multiple Datasets</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Access spending data, vendor payments, contracts, and more
                  through a unified interface.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <FileText className="h-5 w-5 text-accent" />
                  <CardTitle className="text-lg">Audit Trails</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Full transparency with reasoning steps, sources, and
                  methodology for every query.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Center: Hero */}
        <div className="order-1 lg:order-2 lg:col-span-6 flex flex-col items-center text-center space-y-6">
          <Image
            src="/logo1b.png"
            alt="ASTRA logo"
            width={120}
            height={120}
            className="rounded bg-transparent object-contain"
            priority
          />
          <Badge variant="secondary" className="text-xs font-normal">
            Government Data Platform
          </Badge>
          <h1 className="text-5xl font-bold text-foreground">ASTRA</h1>
          <p className="text-sm text-muted-foreground">
            An Accurate and Trustworthy Chatbot for Data Interactions
          </p>
          <p className="text-xs text-muted-foreground max-w-md">
            Interact with government datasets through natural language queries
            while maintaining transparency and trust through audit trails and
            confidence indicators.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/chatbot">
              <Button size="lg" className="w-full sm:w-auto">
                <Search className="mr-2 h-4 w-4" />
                Start Querying Data
              </Button>
            </Link>
            <Button
              variant="outline"
              size="lg"
              className="w-full sm:w-auto bg-transparent"
            >
              <FileText className="mr-2 h-4 w-4" />
              View Documentation
            </Button>
          </div>
        </div>

        {/* Right: Trust Levels */}
        <div className="order-3 lg:col-span-3 space-y-6">
          <h2 className="text-2xl font-bold text-foreground">
            Understanding Trust Levels
          </h2>
          <p className="text-muted-foreground">
            Clear indicators to help you evaluate response reliability
          </p>
          <div className="space-y-4">
            <Card className="border-chart-2">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-chart-2" />
                  <CardTitle className="text-lg text-chart-2">
                    High Trust
                  </CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Verified across multiple sources with high confidence.
                  Suitable for policy and reporting.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-accent">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5 text-accent" />
                  <CardTitle className="text-lg text-accent">
                    Medium Trust
                  </CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Generally reliable with some limitations. Verify for critical
                  use.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="border-destructive">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5 text-destructive" />
                  <CardTitle className="text-lg text-destructive">
                    Low Trust
                  </CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Limited or conflicting data. Use cautiously and verify
                  further.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="text-center space-y-6 py-12 bg-muted rounded-lg mt-16">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">
            Ready to Get Started?
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Begin exploring government data with natural language queries. Our
            chatbot provides transparent, trustworthy insights with full audit
            trails.
          </p>
        </div>
        <Link href="/chatbot">
          <Button size="lg">
            <Search className="mr-2 h-4 w-4" />
            Launch ASTRA Chatbot
          </Button>
        </Link>
      </section>
    </div>
  );
}
