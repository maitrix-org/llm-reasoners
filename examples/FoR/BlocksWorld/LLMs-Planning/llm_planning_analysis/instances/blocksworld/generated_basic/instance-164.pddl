

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b d)
(ontable c)
(ontable d)
(clear a)
(clear b)
(clear c)
)
(:goal
(and
(on c a))
)
)


