(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f k l b)
(:init 
(harmony)
(planet f)
(planet k)
(planet l)
(planet b)
(province f)
(province k)
(province l)
(province b)
)
(:goal
(and
(craves f k)
(craves k l)
(craves l b)
)))