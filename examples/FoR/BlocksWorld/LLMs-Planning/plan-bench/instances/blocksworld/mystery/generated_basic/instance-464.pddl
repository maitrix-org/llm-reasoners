

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a b)
(craves b e)
(craves c a)
(planet d)
(craves e d)
(province c)
)
(:goal
(and
(craves b c)
(craves c d)
(craves d a)
(craves e b))
)
)


