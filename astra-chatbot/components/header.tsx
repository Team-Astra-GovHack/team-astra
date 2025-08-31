"use client";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { usePathname } from "next/navigation";
import Image from "next/image";

export function Header() {
  const pathname = usePathname();
  return (
    <header className="border-b border-border bg-card ">
      <div className="px-4 max-w-7xl mx-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href="/" className="flex items-center space-x-2">
              <Image
                src="/logo1b.png"
                alt="ASTRA logo"
                width={60}
                height={100}
                className="rounded bg-transparent object-contain"
                priority
              />
            </Link>
          </div>
          <nav className="flex items-center space-x-4">
            {pathname !== "/" && (
              <Link href="/">
                <Button variant="default" size="sm">
                  Home
                </Button>
              </Link>
            )}
            {pathname !== "/chatbot" && (
              <Link href="/chatbot">
                <Button variant="default" size="sm">
                  Launch Chatbot
                </Button>
              </Link>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
}
