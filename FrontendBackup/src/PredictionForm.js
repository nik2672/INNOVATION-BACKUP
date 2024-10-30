// PredictionForm.js
import React, { useState } from 'react';
import {
  Container, Box, Typography, Fade, TextField, MenuItem, Button,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

function PredictionForm() {
  const [rooms, setRooms] = useState('');
  const [houseType, setHouseType] = useState('');
  const [postcode, setPostcode] = useState('');
  const [distance, setDistance] = useState('');
  const [postcodeError, setPostcodeError] = useState(false);
  const [postcodeHelperText, setPostcodeHelperText] = useState('');

  const navigate = useNavigate();

  const handleSubmit = (event) => {
    event.preventDefault();

    // Reset error messages
    setPostcodeError(false);
    setPostcodeHelperText('');

    // Validate postcode
    if (postcode.length !== 4) {
      setPostcodeError(true);
      setPostcodeHelperText('Postcode must be exactly 4 digits');
      return; // Prevent form submission
    }

    // Prepare data
    const data = {
      Rooms: parseInt(rooms),
      Type: houseType,
      Postcode: parseInt(postcode),
      Distance: parseFloat(distance),
    };

    console.log("Form data:", data);  // Debug: Check formData before navigation

    // Navigate to PredictionPage with the data
    navigate('/prediction', { state: { formData: data } });
  };

  return (
    <Container component="main" sx={{ mt: 8, mb: 2, flex: 1 }}>
      <Fade in timeout={1000}>
        <Box>
          <Typography variant="h3" component="h1" gutterBottom>
            Real Estate Price Prediction
          </Typography>
          <Typography variant="body1" gutterBottom>
            Enter the details below to get an estimated price for the property.
          </Typography>
          <form onSubmit={handleSubmit}>
            <TextField
              required
              label="Number of Rooms"
              type="number"
              value={rooms}
              onChange={(e) => setRooms(e.target.value)}
              fullWidth
              margin="normal"
            />
            <TextField
              required
              select
              label="House Type"
              value={houseType}
              onChange={(e) => setHouseType(e.target.value)}
              fullWidth
              margin="normal"
            >
              <MenuItem value="h">House</MenuItem>
              <MenuItem value="t">Townhouse</MenuItem>
              <MenuItem value="u">Unit</MenuItem>
            </TextField>
            <TextField
              required
              label="Postcode"
              type="number"
              value={postcode}
              onChange={(e) => setPostcode(e.target.value)}
              error={postcodeError}
              helperText={postcodeHelperText}
              fullWidth
              margin="normal"
            />
            <TextField
              required
              label="Distance from City (km)"
              type="number"
              value={distance}
              onChange={(e) => setDistance(e.target.value)}
              fullWidth
              margin="normal"
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              sx={{ mt: 2 }}
            >
              Predict Price
            </Button>
          </form>
        </Box>
      </Fade>
    </Container>
  );
}

export default PredictionForm;
