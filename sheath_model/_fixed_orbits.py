"""Collisionless orbit moments shared by the fixed-inlet sheath solvers."""

import math

import numpy as np
from scipy.special import erf, erfcx

from ._ions import ion_critical_potential
from ._orbits import unit_gauss


class FixedReservoirOrbits:
    def __init__(self, reservoir, emission, order):
        self.reservoir = reservoir
        self.temperature = emission.temperature_ev
        self.speed = reservoir.speed_unit
        self.density = reservoir.ion_density_m3
        self.source = emission.flux_m2_s / (self.speed * self.density)
        self.ion_flux = reservoir.ion_flux_m2_s / (self.speed * self.density)
        self.ion_limit = ion_critical_potential(reservoir.ion_energy_ev, reservoir.ion_pressure_ev)
        self.x, self.w = unit_gauss(order)
        q, weights = unit_gauss(max(160, 2 * order))
        scale = math.sqrt(max(component.temperature_ev if hasattr(component, "temperature_ev")
                              else component.core_temperature_ev for component in reservoir.electrons))
        self.q, self.weights = scale*q/(1-q), scale*weights/(1-q)**2

    def wind(self, energy, direction):
        return self.reservoir.phase(direction * np.sqrt(np.maximum(energy, 0.0))) / self.density

    def ions(self, potential):
        return self.reservoir.ion_density(potential) / self.density

    def influx(self, depth):
        return self.reservoir.inward_flux(depth) / (self.speed * self.density)

    def escape_for_zero_current(self, depth):
        """Required escaping emission for a collisionless return path."""
        return self.influx(depth) - self.ion_flux

    def passing(self, depth, cutoff, direction):
        low = np.sqrt(np.maximum(cutoff - depth, 0.0))[..., None]
        energy = depth[..., None] + low**2 + self.q*(2*low + self.q)
        return self.wind(energy, direction) @ self.weights

    def panel(self, function, low, high):
        # The sine map removes the square-root cusp at the cutoff.
        a = np.pi * self.x / 2
        widths = (np.asarray(high) - low)[..., None]
        points = np.asarray(low)[..., None] + widths * np.sin(a)**2
        return np.sum(function(points) * widths * np.pi * np.sin(a) * np.cos(a) * self.w, axis=-1)

    def photo(self, energy, depth, amplitude):
        return amplitude * np.exp((depth - energy) / self.temperature)

    def photo_density(self, potential, amplitude, cutoff):
        return amplitude * math.sqrt(math.pi * self.temperature) / 2 * erfcx(
            np.sqrt(np.maximum(cutoff + potential, 0.0) / self.temperature)
        )

    def upstream_edge_excess(self, h, depth, amplitude):
        psi = depth - h
        reflected = self.panel(lambda v: self.wind(psi[..., None] + v*v, 1), 0.0, np.sqrt(h))
        return self.passing(psi, 0.0, 1) + reflected + self.photo_density(-psi, amplitude, depth) - self.ions(-psi)

    def surface_excess(self, h, depth, amplitude):
        t = self.temperature
        return (self.passing(depth-h, depth, 1)
                + amplitude * math.sqrt(math.pi*t)/2 * np.exp(h/t) * (1 + erf(np.sqrt(h/t)))
                - self.ions(h-depth))
