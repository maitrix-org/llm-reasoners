

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a d)
(craves b e)
(craves c a)
(planet d)
(planet e)
(province b)
(province c)
)
(:goal
(and
(craves a b)
(craves b d)
(craves c a))
)
)


