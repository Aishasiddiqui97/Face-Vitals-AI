# 🚀 Vercel Deployment Guide

## Prerequisites

1. **Install Vercel CLI**:
```bash
npm i -g vercel
```

2. **Login to Vercel**:
```bash
vercel login
```

## Deployment Steps

### Method 1: Command Line (Recommended)

1. **Navigate to project directory**:
```bash
cd your-project-folder
```

2. **Deploy to Vercel**:
```bash
vercel --prod
```

3. **Follow the prompts**:
   - Set up and deploy? **Y**
   - Which scope? **Your account**
   - Link to existing project? **N**
   - Project name? **face-vitals-ai**
   - Directory? **./** (current directory)

### Method 2: GitHub Integration

1. **Push to GitHub**:
```bash
git add .
git commit -m "Deploy to Vercel"
git push origin main
```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import from GitHub: `Aishasiddiqui97/Face-Vitals-AI`
   - Deploy

## Project Structure

```
face-vitals-ai/
├── api/
│   ├── app.py              # Flask backend
│   └── facebp_core.py      # Core detection logic
├── templates/
│   └── index.html          # Frontend interface
├── vercel.json             # Vercel configuration
├── requirements_flask.txt   # Python dependencies
└── package.json            # Node.js metadata
```

## Environment Variables (if needed)

```bash
# Set environment variables
vercel env add PYTHON_VERSION 3.9
```

## Custom Domain (Optional)

1. **Add domain in Vercel dashboard**
2. **Update DNS records**:
   - Type: CNAME
   - Name: www (or @)
   - Value: cname.vercel-dns.com

## Troubleshooting

### Common Issues:

1. **Build fails**: Check `requirements_flask.txt` dependencies
2. **Camera not working**: Ensure HTTPS (Vercel provides this automatically)
3. **Slow response**: Optimize image processing in `facebp_core.py`

### Debug Commands:

```bash
# Local development
vercel dev

# Check logs
vercel logs

# Redeploy
vercel --prod --force
```

## Expected URLs

- **Production**: `https://face-vitals-ai.vercel.app`
- **Custom domain**: `https://your-domain.com`

## Performance Notes

- First request may be slow (cold start)
- Subsequent requests will be faster
- Camera processing happens client-side for better performance

## Security

- ✅ HTTPS enabled by default
- ✅ Camera access requires user permission
- ✅ No data stored on server
- ✅ Real-time processing only

## Support

If deployment fails, check:
1. Vercel dashboard for error logs
2. Browser console for client-side errors
3. Network tab for API call failures