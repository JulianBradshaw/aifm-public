import { ParametersType } from '@/constants/parameters';

/**
 * Export parameters to a JSON file
 */
export function exportParametersToFile(parameters: ParametersType): void {
  const parameterData = JSON.stringify(parameters, null, 2);
  const blob = new Blob([parameterData], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'parameter-settings.json';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Import parameters from a JSON file
 * Returns the parsed parameters or null if import failed
 */
export async function importParametersFromFile(file: File): Promise<ParametersType | null> {
  try {
    const fileContents = await file.text();
    const parsed = JSON.parse(fileContents);
    if (!parsed || typeof parsed !== 'object') {
      throw new Error('Invalid parameter file');
    }
    return parsed as ParametersType;
  } catch (error) {
    console.error('Failed to import parameters:', error);
    return null;
  }
}
