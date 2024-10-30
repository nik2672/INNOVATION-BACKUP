// PredictionPage.js
import React, { useState, useEffect } from 'react';
import {
  Container, Box, Typography, Fade, CircularProgress,
} from '@mui/material';
import { useLocation } from 'react-router-dom';

function PredictionPage() {
  const location = useLocation();
  const { formData } = location.state || {};
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [error, setError] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(true);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        // Send data to backend API
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
        });

        if (response.ok) {
          const result = await response.json();
          console.log("Backend response:", result);  // Debug: Check the response from backend
          setError(null);
          setPredictedPrice(result.predicted_price);
        } else {
          setError('Error fetching prediction.');
          setPredictedPrice(null);
        }
      } catch (error) {
        console.error('Error:', error);
        setError('Error fetching prediction.');
        setPredictedPrice(null);
      } finally {
        setLoadingPrediction(false);
      }
    };

    if (formData) {
      fetchPrediction();
    } else {
      setError('No data provided.');
      setLoadingPrediction(false);
    }
  }, [formData]);

  return (
    <Container component="main" sx={{ mt: 8, mb: 2, flex: 1 }}>
      <Fade in timeout={1000}>
        <Box>
          <Typography variant="h3" component="h1" gutterBottom>
            Prediction Result
          </Typography>

          {loadingPrediction && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
              <CircularProgress />
            </Box>
          )}

          {predictedPrice && (
            <Typography variant="h5" sx={{ mt: 4 }}>
              Predicted Price: ${predictedPrice.toFixed(2)}
            </Typography>
          )}

          {error && (
            <Typography color="error" sx={{ mt: 4 }}>
              {error}
            </Typography>
          )}
        </Box>
      </Fade>
    </Container>
  );
}

export default PredictionPage;
