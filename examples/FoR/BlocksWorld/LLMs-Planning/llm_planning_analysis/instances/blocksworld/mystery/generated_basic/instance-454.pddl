

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b c)
(craves c d)
(craves d e)
(craves e a)
(province b)
)
(:goal
(and
(craves a b)
(craves e d))
)
)


