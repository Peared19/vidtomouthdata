export class ApiClient {
  constructor({ baseUrl }) {
    this.baseUrl = baseUrl || '';
  }

  async animate(text) {
    const res = await fetch(`${this.baseUrl}/animate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    let json;
    try {
      json = await res.json();
    } catch {
      throw new Error(`Server returned non-JSON (status ${res.status})`);
    }

    if (!res.ok) {
      throw new Error(json?.error || `Server error (status ${res.status})`);
    }

    return json;
  }
}
