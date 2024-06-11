

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a d)
(ontable b)
(on c a)
(ontable d)
(clear b)
(clear c)
)
(:goal
(and
(on b c)
(on c d))
)
)


