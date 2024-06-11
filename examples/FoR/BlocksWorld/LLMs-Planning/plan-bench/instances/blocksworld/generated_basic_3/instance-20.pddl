

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(on a c)
(ontable b)
(ontable c)
(clear a)
(clear b)
)
(:goal
(and
(on b a))
)
)


