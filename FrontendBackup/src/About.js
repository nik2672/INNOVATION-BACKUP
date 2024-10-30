import React from 'react';
import { Container, Typography, Grid } from '@mui/material';

function About() {
  return (
    <Container>
      {/* Adjusting the About header */}
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 4 }}>
        About
      </Typography>

      {/* Adjusted Our Story header */}
      <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 2 }}>
        Our Story
      </Typography>

      <Grid container spacing={4}>
        <Grid item xs={12} sm={6} display="flex" alignItems="center" justifyContent="flex-end">
          <Typography variant="body1" gutterBottom>
            This project seeks to explore and assess housing prices and market trends across different property types in Melbourne, Australia. By employing machine learning methods, our team will create an AI model designed for prediction, attribution, and classification to improve insights and forecasts of housing market behavior. This research is especially pertinent due to the growing complexity and fluctuations in Melbourne's real estate markets. Made by Swinburne students to advance the search of houses.
          </Typography>
        </Grid>
        <Grid item xs={12} sm={6}>
          <img src="/images/house.jpg" alt="Description of the image" style={{ width: '100%', height: 'auto' }} />
        </Grid>
      </Grid>
      <Grid container spacing={4} sx={{ mt: 4 }}>
        <Grid item xs={12} sm={6}>
          <Typography variant="body1" gutterBottom>
            This tool is designed to offer comprehensive insights into the housing market by investigating and analyzing housing prices and the performance of different types of properties. Leveraging advanced machine learning techniques, it enables users to predict market trends, attribute factors influencing prices, and classify housing data to gain a deeper understanding of market dynamics. By providing detailed analysis and forecasts, this tool empowers users to make informed decisions, whether they are buyers, sellers, investors, or real estate professionals, helping them navigate the complexities of the housing market with greater confidence.
          </Typography>
        </Grid>
        <Grid item xs={12} sm={6} display="flex" alignItems="center" justifyContent="flex-start">
          <Typography variant="body1" gutterBottom>
            Welcome to a world of limitless possibilities, where the journey is as exhilarating as the destination, and where every moment is an opportunity to make your mark on the houses of existence. The only limit is the extent of your imagination.
          </Typography>
        </Grid>
      </Grid>
    </Container>
  );
}

export default About;
