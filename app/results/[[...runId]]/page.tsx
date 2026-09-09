import { redirect } from "next/navigation";
import { Suspense } from "react";

async function ResultsRedirectContent({
  params,
}: {
  params: Promise<{ runId?: string[] }>;
}): Promise<never> {
  const resolvedParams = await params;
  const runId = resolvedParams.runId?.[0];

  if (runId) {
    redirect(`/analysis/${runId}`);
  } else {
    redirect("/analysis");
  }
}

export default function ResultsRedirect({
  params,
}: {
  params: Promise<{ runId?: string[] }>;
}) {
  return (
    <Suspense fallback={null}>
      <ResultsRedirectContent params={params} />
    </Suspense>
  );
}
