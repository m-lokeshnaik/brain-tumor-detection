import React, { useState } from 'react';
import { 
  Container, Box, Typography, Paper, CircularProgress,
  Alert, Button, Card, CardContent, Grid
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

interface Prediction {
  prediction: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  processing_time?: number;
}

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);

  const onDrop = async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file size
    if (file.size > 5 * 1024 * 1024) {
      setError('File size must be less than 5MB');
      return;
    }

    setImage(URL.createObjectURL(file));
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image');
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png']
    },
    multiple: false,
    maxSize: 5 * 1024 * 1024
  });

  const chartData = prediction?.class_probabilities ? 
    Object.entries(prediction.class_probabilities).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      probability: Number(value.toFixed(2))
    })) : [];

  const handleReset = () => {
    setImage(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ mb: 4 }}>
          Brain Tumor Detection
        </Typography>

        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Upload MRI Scan
                </Typography>
                
                <Paper
                  {...getRootProps()}
                  sx={{
                    p: 3,
                    mt: 2,
                    textAlign: 'center',
                    cursor: 'pointer',
                    bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                    border: '2px dashed',
                    borderColor: isDragActive ? 'primary.main' : 'grey.300',
                    '&:hover': {
                      borderColor: 'primary.main'
                    }
                  }}
                >
                  <input {...getInputProps()} />
                  <Typography>
                    {isDragActive
                      ? "Drop the MRI scan here"
                      : "Drag 'n' drop an MRI scan, or click to select"}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                    Supported formats: JPG, PNG (max 5MB)
                  </Typography>
                </Paper>

                {error && (
                  <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                  </Alert>
                )}

                {loading && (
                  <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                    <CircularProgress />
                  </Box>
                )}

                {image && (
                  <Box sx={{ mt: 3, textAlign: 'center' }}>
                    <img 
                      src={image} 
                      alt="MRI Scan"
                      style={{ 
                        maxWidth: '100%', 
                        maxHeight: '400px',
                        borderRadius: '8px',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                      }}
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            {prediction && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Analysis Results
                  </Typography>

                  <Box sx={{ mb: 3 }}>
                    <Typography variant="h5" color="primary" gutterBottom>
                      {prediction.prediction.charAt(0).toUpperCase() + prediction.prediction.slice(1)}
                    </Typography>
                    <Typography variant="body1" gutterBottom>
                      Confidence: {prediction.confidence.toFixed(2)}%
                    </Typography>
                    {prediction.processing_time && (
                      <Typography variant="body2" color="textSecondary">
                        Processing Time: {prediction.processing_time.toFixed(2)}s
                      </Typography>
                    )}
                  </Box>

                  <Typography variant="h6" sx={{ mb: 2 }}>
                    Class Probabilities
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={60} />
                        <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Bar dataKey="probability" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleReset}
            disabled={!image}
            sx={{ mr: 2 }}
          >
            Reset
          </Button>
        </Box>
      </Box>
    </Container>
  );
}

export default App;