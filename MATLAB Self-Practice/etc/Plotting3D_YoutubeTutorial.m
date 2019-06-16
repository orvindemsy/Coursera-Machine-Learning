clear;
clc;
close all;

%%3D Surface
%patch
x = [1 2 5];
y = [2 3 4];
z = [1 3 0];
figure(1);
patch(x, y, z, 'm');
grid on;

%%Applying 3D ellips function
t = linspace(0, 6*pi, 30);
x1 = 3* cos(t);
y1 = 1* sin(t);
z1 = 0.01* t .^2;


figure(2)
plot3(x1, y1, z1);
hold on;
plot3(x1, y1, z1,'mo');
xlabel('x');
ylabel('y');
zlabel('z');
grid on;
axis('equal');
hold off;


%%Scatter function holds the same function as plot with mo markers
%{
figure
scatter3(x, y, z);
xlabel('x');
ylabel('y');
zlabel('z');
grid on;
axis('equal');
%}

%%Mesh Function
x2 = linspace(-pi, pi, 20);
x3 = linspace(-10, 16, 30);

[X1, X2] = meshgrid(x2, x3);

%Evaluate the function at these (X1, X2) pairs
Z = cos(X1).*X2;

figure(3);
mesh(X1, X2, Z)
xlabel('x_1');
ylabel('x_2');
zlabel('z = f(x_1, x_2)');
grid on;
title('Using the ''mesh'' function');

%%Surf Function
x2 = linspace(-pi, pi, 20);
x3 = linspace(-10, 16, 30);

[X1, X2] = meshgrid(x2, x3);

%Evaluate the function at these (X1, X2) pairs
Z = cos(X1).*X2;

figure(4);
surf(X1, X2, Z)
xlabel('x_1');
ylabel('x_2');
zlabel('z = f(x_1, x_2)');
grid on;
title('Using the ''surf'' function');
shading interp %get rid of black lines in the surface plot

colorbar %adds a colorbar that acts as legend for colors

%%Contour
x2 = linspace(-pi, pi, 20);
x3 = linspace(-10, 16, 30);

[X1, X2] = meshgrid(x2, x3);

%Evaluate the function at these (X1, X2) pairs
Z = cos(X1).*X2;

figure(5);
contour(X1, X2, Z)
xlabel('x_1');
ylabel('x_2');
zlabel('z = f(x_1, x_2)');
grid on;
title('Using the ''contour'' function');

%%Surface Contour
x2 = linspace(-pi, pi, 20);
x3 = linspace(-10, 16, 30);

[X1, X2] = meshgrid(x2, x3);

%Evaluate the function at these (X1, X2) pairs
Z = cos(X1).*X2;

figure(6);
surfc(X1, X2, Z)
xlabel('x_1');
ylabel('x_2');
zlabel('z = f(x_1, x_2)');
grid on;
title('Using the ''contour'' function');