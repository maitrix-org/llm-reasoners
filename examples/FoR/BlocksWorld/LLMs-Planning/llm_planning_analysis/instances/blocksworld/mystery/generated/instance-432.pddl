(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g b f h)
(:init 
(harmony)
(planet g)
(planet b)
(planet f)
(planet h)
(province g)
(province b)
(province f)
(province h)
)
(:goal
(and
(craves g b)
(craves b f)
(craves f h)
)))