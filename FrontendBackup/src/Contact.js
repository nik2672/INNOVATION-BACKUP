// Contact.js
import React from 'react';
import { Container, Typography, Grid, Fade, Card, CardContent, CardMedia } from '@mui/material';

// our contact page listing our team members
function Contact() {
  const teamMembers = [
    {
      name: 'Ibrahim M Barmil',
      role: 'Team Member',
      email: '104490388@student.swin.edu.au',
      image: '/images/pic1.JPG',
    },

    {
      name: 'Nikhil Mohite',
      role: 'Team Member',
      email: '104549772@student.swin.edu.au',
      image: '/images/pic2.JPEG',
    },

    {
      name: 'Ryan Horsfall',
      role: 'Team Member',
      email: '103098655@student.swin.edu.au',
      image: '/images/pic3.jpg',
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

export default Contact;
