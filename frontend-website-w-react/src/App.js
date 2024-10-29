// App.js
import React, { useState, useMemo } from 'react';
import {
  AppBar, Toolbar, Typography, Container, Button, Box,
  Drawer, List, ListItem, ListItemIcon, ListItemText, IconButton,
  Switch, Snackbar, Alert, Divider, TextField, MenuItem,
  Card, CardContent, CardMedia, Grid, Fade, CircularProgress,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home as HomeIcon,
  Info as InfoIcon,
  Mail as MailIcon,
  Brightness4 as Brightness4Icon,
  Brightness7 as Brightness7Icon,
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { red } from '@mui/material/colors';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';

// About page component
function About() {
  return (
    <Container>
      <Typography variant="h3" component="h1" gutterBottom>
        About Us
      </Typography>
      <Typography variant="body1" gutterBottom>
        Welcome to our Real Estate Price Predictor application. Our team is dedicated to providing accurate and reliable price predictions to help you make informed decisions in the real estate market.
      </Typography>
    </Container>
  );
}

// Contact page component
function Contact() {
  const teamMembers = [
    {
      name: 'John Doe',
      role: 'Data Scientist',
      email: 'john.doe@example.com',
      image: 'https://via.placeholder.com/150',
    },
    {
      name: 'Jane Smith',
      role: 'Frontend Developer',
      email: 'jane.smith@example.com',
      image: 'https://via.placeholder.com/150',
    },
    {
      name: 'Bob Johnson',
      role: 'Backend Developer',
      email: 'bob.johnson@example.com',
      image: 'https://via.placeholder.com/150',
    },
  ];

  return (
    <Container>
      <Typography variant="h3" component="h1" gutterBottom>
        Contact Us
      </Typography>
      <Grid container spacing={4}>
        {teamMembers.map((member, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Fade in timeout={1000}>
              <Card>
                <CardMedia
                  component="img"
                  height="200"
                  image={member.image}
                  alt={member.name}
                />
                <CardContent>
                  <Typography variant="h5" component="div">
                    {member.name}
                  </Typography>
                  <Typography variant="subtitle1" color="text.secondary">
                    {member.role}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {member.email}
                  </Typography>
                </CardContent>
              </Card>
            </Fade>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

// Prediction Form Component
function PredictionForm() {
  const [rooms, setRooms] = useState('');
  const [houseType, setHouseType] = useState('');
  const [postcode, setPostcode] = useState('');
  const [distance, setDistance] = useState('');
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [error, setError] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);

  // Validation states
  const [postcodeError, setPostcodeError] = useState(false);
  const [postcodeHelperText, setPostcodeHelperText] = useState('');

  const handleSubmit = async (event) => {
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

    try {
      setLoadingPrediction(true);
      // Send data to backend API
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const result = await response.json();
        setPredictedPrice(result.predicted_price);
        setError(null);
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
              disabled={loadingPrediction}
            >
              {loadingPrediction ? <CircularProgress size={24} /> : 'Predict Price'}
            </Button>
          </form>

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

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);

  // Create a theme that uses the dark mode setting and sets primary color to red
  const theme = useMemo(() => createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: red[500],
      },
    },
  }), [darkMode]);

  const toggleDrawer = (open) => (event) => {
    if (
      event &&
      event.type === 'keydown' &&
      (event.key === 'Tab' || event.key === 'Shift')
    ) {
      return;
    }
    setDrawerOpen(open);
  };

  const handleDarkModeToggle = () => {
    setDarkMode(!darkMode);
    setSnackbarOpen(true);
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  const drawerContent = (
    <Box
      sx={{ width: 250 }}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <List>
        <ListItem button component={Link} to="/">
          <ListItemIcon><HomeIcon /></ListItemIcon>
          <ListItemText primary="Home" />
        </ListItem>
        <ListItem button component={Link} to="/about">
          <ListItemIcon><InfoIcon /></ListItemIcon>
          <ListItemText primary="About" />
        </ListItem>
        <ListItem button component={Link} to="/contact">
          <ListItemIcon><MailIcon /></ListItemIcon>
          <ListItemText primary="Contact" />
        </ListItem>
      </List>
      <Divider />
      <List>
        <ListItem>
          <ListItemIcon>
            {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
          </ListItemIcon>
          <ListItemText primary="Dark Mode" />
          <Switch checked={darkMode} onChange={handleDarkModeToggle} />
        </ListItem>
      </List>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <AppBar position="static" color="primary" enableColorOnDark>
            <Toolbar>
              <IconButton
                edge="start"
                color="inherit"
                aria-label="menu"
                onClick={toggleDrawer(true)}
              >
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Real Estate Price Predictor
              </Typography>
              <Button color="inherit" component={Link} to="/about">
                About
              </Button>
              <Button color="inherit" component={Link} to="/contact">
                Contact
              </Button>
            </Toolbar>
          </AppBar>

          <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer(false)}>
            {drawerContent}
          </Drawer>

          <Routes>
            <Route path="/" element={<PredictionForm />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
          </Routes>

          <Box
            component="footer"
            sx={{ bgcolor: 'background.paper', py: 6, mt: 'auto' }}
          >
            <Container maxWidth="lg">
              <Typography variant="body1" align="center">
                Real Estate Price Predictor
              </Typography>
              <Typography variant="body2" color="text.secondary" align="center">
                {'© '}
                {new Date().getFullYear()} High Distinction Only Group 14 - 05
              </Typography>
            </Container>
          </Box>

          <Snackbar
            open={snackbarOpen}
            autoHideDuration={6000}
            onClose={handleSnackbarClose}
          >
            <Alert
              onClose={handleSnackbarClose}
              severity="success"
              sx={{ width: '100%' }}
            >
              {darkMode ? 'Dark mode enabled!' : 'Light mode enabled!'}
            </Alert>
          </Snackbar>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
