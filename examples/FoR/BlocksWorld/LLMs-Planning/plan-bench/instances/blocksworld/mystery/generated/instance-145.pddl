(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h j f l)
(:init 
(harmony)
(planet h)
(planet j)
(planet f)
(planet l)
(province h)
(province j)
(province f)
(province l)
)
(:goal
(and
(craves h j)
(craves j f)
(craves f l)
)))