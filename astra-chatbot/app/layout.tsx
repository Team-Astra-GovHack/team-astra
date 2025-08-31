import type React from "react";
import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import ReactQueryProvider from "@/components/react-query-provider";
import { Header } from "@/components/header";
import { SkipLink } from "@/components/accessibility/skip-link";
import { ErrorBoundary } from "@/components/error-boundary";
import { Suspense } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASTRA - Accurate and Trustworthy Data Chatbot",
  description:
    "Government data interaction platform with trust indicators and audit trails",
  generator: "v0.app",
  keywords:
    "government data, chatbot, transparency, audit trails, public information",
  authors: [{ name: "Government Technology Office" }],
  viewport: "width=device-width, initial-scale=1",
  robots: "index, follow",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`font-sans ${GeistSans.variable} ${GeistMono.variable} antialiased `}
      >
        <SkipLink />
        <ReactQueryProvider>
          <ErrorBoundary>
            <Suspense
              fallback={<div className="sr-only">Loading application...</div>}
            >
              <div className="min-h-screen bg-background">
                {/* <Header /> */}
                <main
                  id="main-content"
                  className="container mx-auto px-4 py-8"
                  tabIndex={-1}
                >
                  {children}
                </main>
              </div>
            </Suspense>
          </ErrorBoundary>
        </ReactQueryProvider>
      </body>
    </html>
  );
}
