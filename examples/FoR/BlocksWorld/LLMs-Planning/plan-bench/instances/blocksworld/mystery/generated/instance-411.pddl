(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h i k j)
(:init 
(harmony)
(planet h)
(planet i)
(planet k)
(planet j)
(province h)
(province i)
(province k)
(province j)
)
(:goal
(and
(craves h i)
(craves i k)
(craves k j)
)))