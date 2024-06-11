(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k f j h)
(:init 
(harmony)
(planet k)
(planet f)
(planet j)
(planet h)
(province k)
(province f)
(province j)
(province h)
)
(:goal
(and
(craves k f)
(craves f j)
(craves j h)
)))