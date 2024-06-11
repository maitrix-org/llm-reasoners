(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f c k h)
(:init 
(harmony)
(planet f)
(planet c)
(planet k)
(planet h)
(province f)
(province c)
(province k)
(province h)
)
(:goal
(and
(craves f c)
(craves c k)
(craves k h)
)))