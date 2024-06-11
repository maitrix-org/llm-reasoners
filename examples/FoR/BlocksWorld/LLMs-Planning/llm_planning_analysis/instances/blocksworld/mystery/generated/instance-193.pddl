(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k j b)
(:init 
(harmony)
(planet k)
(planet j)
(planet b)
(province k)
(province j)
(province b)
)
(:goal
(and
(craves k j)
(craves j b)
)))