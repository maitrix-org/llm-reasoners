(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h j l)
(:init 
(harmony)
(planet h)
(planet j)
(planet l)
(province h)
(province j)
(province l)
)
(:goal
(and
(craves h j)
(craves j l)
)))