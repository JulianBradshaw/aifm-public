import cloudinary from 'cloudinary';
import { parseHTML } from 'linkedom';
import crypto from 'crypto';

cloudinary.v2.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
})

function getHashableUrl(url: string) {
  try {
    const urlObj = new URL(url);
    urlObj.hash = '';
    // Images hosted on *.googleusercontent.com have an access key in a query parameter,
    // which might not be stable over time even if the image hasn't changed.  Since we
    // don't want to re-upload the same image if the key changes, we need to remove it
    // before hashing the url.
    if (urlObj.host.includes('googleusercontent.com')) {
      urlObj.searchParams.delete('key');
    }
    return urlObj.toString();
  } catch (error) {
    console.warn(`Failed to get hashable url for ${url}:`, error);
    return url;
  }
}

export async function uploadGoogleDocImagesToCloudinary(html: string) {
  const { document } = parseHTML(html);

  const images = Array.from(document.querySelectorAll('img'));
  
  if (images.length === 0) {
    return html;
  }
  
  await Promise.all(
    images.map(async (img) => {
      const originalUrl = img.getAttribute('src');
      if (!originalUrl) return;
      
      try {
        let normalizedUrl: string;
        
        if (originalUrl.startsWith('data:')) {
          normalizedUrl = originalUrl;
        } else {
          normalizedUrl = getHashableUrl(originalUrl);
        }
        
        const hashedUrl = crypto.createHash('sha256').update(normalizedUrl).digest('hex');
        
        // We upload the original url rather than the normalized url because the normalized url
        // might not contain an access key (i.e. if it's an image hosted on *.googleusercontent.com)
        const result = await cloudinary.v2.uploader.upload(originalUrl, {
          public_id: hashedUrl,
          overwrite: false,
          asset_folder: 'ai-futures-calculator',
          timeout: 10000, // 10 second timeout per image
        });
        
        img.setAttribute('src', result.secure_url);
      } catch (error) {
        console.warn(`Failed to upload image ${originalUrl.substring(0, 100)}:`, error);
      }
    })
  );
  
  return document.toString();
}
