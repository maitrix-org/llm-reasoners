(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f e k)
(:init 
(harmony)
(planet f)
(planet e)
(planet k)
(province f)
(province e)
(province k)
)
(:goal
(and
(craves f e)
(craves e k)
)))