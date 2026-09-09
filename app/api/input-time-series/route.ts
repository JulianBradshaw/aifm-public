import { NextResponse } from 'next/server';
import path from 'node:path';
import { readFile } from 'node:fs/promises';

export interface InputTimeSeriesDataPoint {
  time: number;
  L_HUMAN: number;
  inference_H100e: number;
  experiment_H100e: number;
  training_H100e: number;
  // Optional display-only series (present when the CSV carries the columns)
  world_H100e?: number;
  total_ai_company_H100e?: number;
}

export interface InputTimeSeriesResponse {
  data: InputTimeSeriesDataPoint[];
  source: string;
}

export async function GET(): Promise<NextResponse> {
  try {
    // Load the default input series from inputs/input_data.csv
    const dataPath = path.join(process.cwd(), 'inputs', 'input_data.csv');
    const raw = await readFile(dataPath, 'utf8');

    const lines = raw
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    const [header, ...dataLines] = lines;
    const headers = header.split(',');

    // Find column indices
    const timeIdx = headers.indexOf('time');
    const lHumanIdx = headers.indexOf('L_HUMAN');
    const inferenceIdx = headers.indexOf('inference_H100e');
    const experimentIdx = headers.indexOf('experiment_H100e');
    const trainingIdx = headers.indexOf('training_H100e');
    const worldIdx = headers.indexOf('world_H100e');
    const companyTotalIdx = headers.indexOf('total_ai_company_H100e');

    if (timeIdx === -1) {
      return NextResponse.json({ error: 'Missing time column' }, { status: 500 });
    }

    const data: InputTimeSeriesDataPoint[] = [];

    for (const line of dataLines) {
      const values = line.split(',');
      const time = Number.parseFloat(values[timeIdx]);

      if (!Number.isFinite(time)) continue;

      const point: InputTimeSeriesDataPoint = {
        time,
        L_HUMAN: lHumanIdx >= 0 ? Number.parseFloat(values[lHumanIdx]) || 0 : 0,
        inference_H100e: inferenceIdx >= 0 ? Number.parseFloat(values[inferenceIdx]) || 0 : 0,
        experiment_H100e: experimentIdx >= 0 ? Number.parseFloat(values[experimentIdx]) || 0 : 0,
        training_H100e: trainingIdx >= 0 ? Number.parseFloat(values[trainingIdx]) || 0 : 0,
      };

      if (worldIdx >= 0) point.world_H100e = Number.parseFloat(values[worldIdx]) || 0;
      if (companyTotalIdx >= 0) point.total_ai_company_H100e = Number.parseFloat(values[companyTotalIdx]) || 0;

      data.push(point);
    }

    const response: InputTimeSeriesResponse = {
      data,
      source: 'inputs/input_data.csv',
    };

    return NextResponse.json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: `Failed to load input time series data: ${message}` }, { status: 500 });
  }
}
